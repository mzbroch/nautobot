from collections import defaultdict

from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import models
from django.db.models import Sum
from django.urls import reverse
from django.utils.functional import classproperty

from nautobot.dcim.choices import CableLengthUnitChoices, CableTypeChoices, CableEndpointSideChoices
from nautobot.dcim.constants import CABLE_TERMINATION_MODELS, COMPATIBLE_TERMINATION_TYPES, NONCONNECTABLE_IFACE_TYPES

from nautobot.dcim.fields import JSONPathField
from nautobot.dcim.utils import (
    decompile_path_node,
    object_to_path_node,
    path_node_to_object,
)
from nautobot.extras.models import Status, StatusModel
from nautobot.extras.utils import extras_features
from nautobot.core.models.generics import BaseModel, PrimaryModel
from nautobot.utilities.fields import ColorField
from nautobot.utilities.utils import to_meters
from .devices import Device
from .device_components import FrontPort, RearPort
from nautobot.utilities.querysets import RestrictedQuerySet


__all__ = (
    "Cable",
    "CablePath",
)


#
# Cables
#


@extras_features(
    "custom_fields",
    "custom_links",
    "custom_validators",
    "export_templates",
    "graphql",
    "relationships",
    "statuses",
    "webhooks",
)
class Cable(PrimaryModel, StatusModel):
    """
    A physical connection between two endpoints.
    """
    type = models.CharField(max_length=50, choices=CableTypeChoices, blank=True)
    label = models.CharField(max_length=100, blank=True)
    color = ColorField(blank=True)
    length = models.PositiveSmallIntegerField(blank=True, null=True)
    length_unit = models.CharField(
        max_length=50,
        choices=CableLengthUnitChoices,
        blank=True,
    )
    # Stores the normalized length (in meters) for database ordering
    _abs_length = models.DecimalField(max_digits=10, decimal_places=4, blank=True, null=True)

    csv_headers = [
        "type",
        "status",
        "label",
        "color",
        "length",
        "length_unit",
    ]

    @property
    def termination_a_type(self):
        termination_a = self.endpoints.filter(side=CableEndpointSideChoices.SIDE_A).first()
        a_type = termination_a.termination._meta.model if termination_a else None

        return a_type

    @property
    def termination_b_type(self):
        termination_b = self.endpoints.filter(side=CableEndpointSideChoices.SIDE_Z).first()
        b_type = termination_b.termination._meta.model if termination_b else None

        return b_type

    @property
    def a_terminations(self):
        if hasattr(self, '_a_terminations'):
            return self._a_terminations
        # Query self.terminations.all() to leverage cached results
        return [
            endpoint.termination for endpoint in self.endpoints.all() if endpoint.side == CableEndpointSideChoices.SIDE_A
        ]

    @a_terminations.setter
    def a_terminations(self, value):
        self._terminations_modified = True
        self._a_terminations = value

    @property
    def b_terminations(self):
        if hasattr(self, '_b_terminations'):
            return self._b_terminations
        # Query self.terminations.all() to leverage cached results
        return [
            endpoint.termination for endpoint in self.endpoints.all() if endpoint.side == CableEndpointSideChoices.SIDE_Z
        ]

    @b_terminations.setter
    def b_terminations(self, value):
        self._terminations_modified = True
        self._b_terminations = value

    def __init__(self, *args, a_terminations=None, b_terminations=None, **kwargs):
        super().__init__(*args, **kwargs)

        # A copy of the PK to be used by __str__ in case the object is deleted
        self._pk = self.pk

        # Cache the original status so we can check later if it's been changed
        self._orig_status = self.status

        self._terminations_modified = False

        # Assign or retrieve A/B terminations
        if a_terminations:
            self.a_terminations = a_terminations
        if b_terminations:
            self.b_terminations = b_terminations

    def __str__(self):
        pk = self.pk or self._pk
        return self.label or f"#{pk}"

    def get_absolute_url(self):
        return reverse("dcim:cable", args=[self.pk])

    @classproperty
    def STATUS_CONNECTED(cls):
        """Return a cached "connected" `Status` object for later reference."""
        if getattr(cls, "__status_connected", None) is None:
            cls.__status_connected = Status.objects.get_for_model(Cable).get(slug="connected")
        return cls.__status_connected

    def clean(self):
        super().clean()

        # Validate length and length_unit
        if self.length is not None and not self.length_unit:
            raise ValidationError("Must specify a unit when setting a cable length")
        elif self.length is None:
            self.length_unit = ''

        if (not self.present_in_database) and (not self.a_terminations or not self.b_terminations):
            raise ValidationError("Must define A and B terminations when creating a new cable.")

        if self._terminations_modified:

            # Check that all termination objects for either end are of the same type
            for terms in (self.a_terminations, self.b_terminations):
                if len(terms) > 1 and not all(isinstance(t, type(terms[0])) for t in terms[1:]):
                    raise ValidationError("Cannot connect different termination types to same end of cable.")

            # Check that termination types are compatible
            if self.a_terminations and self.b_terminations:
                a_type = self.a_terminations[0]._meta.model_name
                b_type = self.b_terminations[0]._meta.model_name
                if b_type not in COMPATIBLE_TERMINATION_TYPES.get(a_type):
                    raise ValidationError(f"Incompatible termination types: {a_type} and {b_type}")

            # Run clean() on any new CableTerminations
            for termination in self.a_terminations:
                CableEndpoint(
                    cable=self,
                    cable_end=CableEndpointSideChoices.SIDE_A,
                    termination=termination
                ).clean()

            for termination in self.b_terminations:
                CableEndpoint(
                    cable=self,
                    cable_end=CableEndpointSideChoices.SIDE_Z,
                    termination=termination
                ).clean()

    def save(self, *args, **kwargs):

        # Store the given length (if any) in meters for use in database ordering
        if self.length and self.length_unit:
            self._abs_length = to_meters(self.length, self.length_unit)
        else:
            self._abs_length = None

        super().save(*args, **kwargs)

        # Update the private pk used in __str__ in case this is a new object (i.e. just got its pk)
        self._pk = self.pk

        # Retrieve existing A/B terminations for the Cable
        a_terminations = {ct.termination: ct for ct in self.endpoints.filter(side=CableEndpointSideChoices.SIDE_A)}  # TODO(mzb) rename to endpoints
        b_terminations = {ct.termination: ct for ct in self.endpoints.filter(side=CableEndpointSideChoices.SIDE_Z)}

        # Delete stale CableTerminations
        if self._terminations_modified:
            for termination, ct in a_terminations.items():
                if termination.pk and termination not in self.a_terminations:
                    ct.delete()
            for termination, ct in b_terminations.items():
                if termination.pk and termination not in self.b_terminations:
                    ct.delete()

        # Save new CableTerminations (if any)
        if self._terminations_modified:
            for termination in self.a_terminations:
                if not termination.present_in_database or termination not in a_terminations:
                    CableEndpoint(cable=self, side=CableEndpointSideChoices.SIDE_A, termination=termination).save()
            for termination in self.b_terminations:
                if not termination.present_in_database or termination not in b_terminations:
                    CableEndpoint(cable=self, side=CableEndpointSideChoices.SIDE_Z, termination=termination).save()

        # trace_paths.send(Cable, instance=self, created=_created)  # TODO(mzb)

    def to_csv(self):
        return (
            # "{}.{}".format(self.termination_a_type.app_label, self.termination_a_type.model),
            # self.termination_a_id,
            # "{}.{}".format(self.termination_b_type.app_label, self.termination_b_type.model),
            # self.termination_b_id,
            self.get_type_display(),
            self.get_status_display(),
            self.label,
            self.color,
            self.length,
            self.length_unit,
        )

    def get_compatible_types(self):
        """
        Return all termination types compatible with termination A.
        """
        if self.termination_a is None:
            return
        return COMPATIBLE_TERMINATION_TYPES[self.termination_a._meta.model_name]


class CableEndpoint(BaseModel):
    """
    Cable Ends.
    """
    cable = models.ForeignKey(
        to="dcim.Cable",
        on_delete=models.CASCADE,
        related_name="endpoints",
        # blank=True, # TODO(mzb)
        # null=True, # TODO(mzb)
    )
    side = models.CharField(
        max_length=1,
        choices=CableEndpointSideChoices,
        # blank=True
    )
    termination_type = models.ForeignKey(
        to=ContentType,
        limit_choices_to=CABLE_TERMINATION_MODELS,
        on_delete=models.PROTECT,
        related_name="+",
        # blank=True, # TODO(mzb)
        # null=True,  # TODO(mzb)
    )
    termination_id = models.UUIDField(
        # blank=True,
        # null=True,
    )
    termination = GenericForeignKey(ct_field="termination_type", fk_field="termination_id")

    objects = RestrictedQuerySet.as_manager()

    class Meta:
        ordering = ('cable', 'side')
        unique_together = (('termination_type', 'termination_id'),)  # TODO(mzb) ensure

    def __str__(self):
        return f"Cable {self.cable} to {self.termination}"

    def clean(self):
        super().clean()

        # Validate interface type (if applicable)
        if self.termination_type.model == 'interface' and self.termination.type in NONCONNECTABLE_IFACE_TYPES:
            raise ValidationError({
                'termination': f'Cables cannot be terminated to {self.termination.get_type_display()} interfaces'
            })

        # A CircuitTermination attached to a ProviderNetwork cannot have a Cable
        if self.termination_type.model == 'circuittermination' and self.termination.provider_network is not None:
            raise ValidationError({
                'termination': "Circuit terminations attached to a provider network may not be cabled."
            })

    def save(self, *args, **kwargs):

        # # Cache objects associated with the terminating object (for filtering)
        # self.cache_related_objects()

        super().save(*args, **kwargs)

        termination_model = self.termination._meta.model
        termination_model.objects.filter(pk=self.termination_id).update(  # TODO(mzb): Caching implications of .update
            cable=self.cable,
            cable_side=self.side,
        )

    def delete(self, *args, **kwargs):

        termination_model = self.termination._meta.model
        termination_model.objects.filter(pk=self.termination_id).update(  # TODO(mzb): Caching implications of .update
            cable=None,
            cable_side=""
        )

        super().delete(*args, **kwargs)


@extras_features("graphql")
class CablePath(BaseModel):
    """
    A CablePath instance represents the physical path from an origin to a destination, including all intermediate
    elements in the path. Every instance must specify an `origin`, whereas `destination` may be null (for paths which do
    not terminate on a PathEndpoint).

    `path` contains a list of nodes within the path, each represented by a tuple of (type, ID). The first element in the
    path must be a Cable instance, followed by a pair of pass-through ports. For example, consider the following
    topology:

                     1                              2                              3
        Interface A --- Front Port A | Rear Port A --- Rear Port B | Front Port B --- Interface B

    This path would be expressed as:

    CablePath(
        origin = Interface A
        destination = Interface B
        path = [Cable 1, Front Port A, Rear Port A, Cable 2, Rear Port B, Front Port B, Cable 3]
    )

    `is_active` is set to True only if 1) `destination` is not null, and 2) every Cable within the path has a status of
    "connected".
    """

    origin_type = models.ForeignKey(to=ContentType, on_delete=models.CASCADE, related_name="+")
    origin_id = models.UUIDField()
    origin = GenericForeignKey(ct_field="origin_type", fk_field="origin_id")
    destination_type = models.ForeignKey(
        to=ContentType,
        on_delete=models.CASCADE,
        related_name="+",
        blank=True,
        null=True,
    )
    destination_id = models.UUIDField(blank=True, null=True)
    destination = GenericForeignKey(ct_field="destination_type", fk_field="destination_id")
    path = JSONPathField()
    is_active = models.BooleanField(default=False)
    is_split = models.BooleanField(default=False)

    class Meta:
        unique_together = ("origin_type", "origin_id")

    def __str__(self):
        status = " (active)" if self.is_active else " (split)" if self.is_split else ""
        return f"Path #{self.pk}: {self.origin} to {self.destination} via {len(self.path)} nodes{status}"

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        # Record a direct reference to this CablePath on its originating object
        model = self.origin._meta.model
        model.objects.filter(pk=self.origin.pk).update(_path=self.pk)

    @property
    def segment_count(self):
        total_length = 1 + len(self.path) + (1 if self.destination else 0)
        return int(total_length / 3)

    @classmethod
    def from_origin(cls, origin):
        """
        Create a new CablePath instance as traced from the given path origin.
        """
        if origin is None or origin.cable is None:
            return None

        # Import added here to avoid circular imports with Cable.
        from nautobot.circuits.models import CircuitTermination

        destination = None
        path = []
        position_stack = []
        is_active = True
        is_split = False

        node = origin
        visited_nodes = set()
        while node.cable is not None:
            if node.id in visited_nodes:
                raise ValidationError("a loop is detected in the path")
            visited_nodes.add(node.id)
            if node.cable.status != Cable.STATUS_CONNECTED:
                is_active = False

            # Follow the cable to its far-end termination
            path.append(object_to_path_node(node.cable))
            peer_termination = node.get_cable_peer()

            # Follow a FrontPort to its corresponding RearPort
            if isinstance(peer_termination, FrontPort):
                path.append(object_to_path_node(peer_termination))
                node = peer_termination.rear_port
                if node.positions > 1:
                    position_stack.append(peer_termination.rear_port_position)
                path.append(object_to_path_node(node))

            # Follow a RearPort to its corresponding FrontPort (if any)
            elif isinstance(peer_termination, RearPort):
                path.append(object_to_path_node(peer_termination))

                # Determine the peer FrontPort's position
                if peer_termination.positions == 1:
                    position = 1
                elif position_stack:
                    position = position_stack.pop()
                else:
                    # No position indicated: path has split, so we stop at the RearPort
                    is_split = True
                    break

                try:
                    node = FrontPort.objects.get(rear_port=peer_termination, rear_port_position=position)
                    path.append(object_to_path_node(node))
                except ObjectDoesNotExist:
                    # No corresponding FrontPort found for the RearPort
                    break

            # Follow a Circuit Termination if there is a corresponding Circuit Termination
            # Side A and Side Z exist
            elif isinstance(peer_termination, CircuitTermination):
                node = peer_termination.get_peer_termination()
                # A Circuit Termination does not require a peer.
                if node is None:
                    destination = peer_termination
                    break
                path.append(object_to_path_node(peer_termination))
                path.append(object_to_path_node(node))

            # Anything else marks the end of the path
            else:
                destination = peer_termination
                break

        if destination is None:
            is_active = False

        return cls(
            origin=origin,
            destination=destination,
            path=path,
            is_active=is_active,
            is_split=is_split,
        )

    def get_path(self):
        """
        Return the path as a list of prefetched objects.
        """
        # Compile a list of IDs to prefetch for each type of model in the path
        to_prefetch = defaultdict(list)
        for node in self.path:
            ct_id, object_id = decompile_path_node(node)
            to_prefetch[ct_id].append(object_id)

        # Prefetch path objects using one query per model type. Prefetch related devices where appropriate.
        prefetched = {}
        for ct_id, object_ids in to_prefetch.items():
            model_class = ContentType.objects.get_for_id(ct_id).model_class()
            queryset = model_class.objects.filter(pk__in=object_ids)
            if hasattr(model_class, "device"):
                queryset = queryset.prefetch_related("device")
            prefetched[ct_id] = {obj.id: obj for obj in queryset}

        # Replicate the path using the prefetched objects.
        path = []
        for node in self.path:
            ct_id, object_id = decompile_path_node(node)
            path.append(prefetched[ct_id][object_id])

        return path

    def get_total_length(self):
        """
        Return the sum of the length of each cable in the path.
        """
        cable_ids = [
            # Starting from the first element, every third element in the path should be a Cable
            decompile_path_node(self.path[i])[1]
            for i in range(0, len(self.path), 3)
        ]
        return Cable.objects.filter(id__in=cable_ids).aggregate(total=Sum("_abs_length"))["total"]

    def get_split_nodes(self):
        """
        Return all available next segments in a split cable path.
        """
        rearport = path_node_to_object(self.path[-1])
        return FrontPort.objects.filter(rear_port=rearport)
